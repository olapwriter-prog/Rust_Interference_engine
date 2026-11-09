fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match rie_server::parse_args(&args) {
        Ok(cfg) => {
            if let Err(e) = rie_server::serve(cfg) {
                eprintln!("rie-server: {e}");
                std::process::exit(1);
            }
        }
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    }
}
